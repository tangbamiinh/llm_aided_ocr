import os
import traceback
import asyncio
import re
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import List, Tuple, Optional
from pdf2image import convert_from_path
import pytesseract
import tiktoken
import numpy as np
from PIL import Image
from decouple import Config as DecoupleConfig, RepositoryEnv, RepositoryEmpty
import cv2
from transformers import AutoTokenizer

# Configuration
# Try to use .env file if it exists, otherwise use environment variables only
try:
    config = DecoupleConfig(RepositoryEnv('.env'))
except FileNotFoundError:
    # If .env doesn't exist, use RepositoryEmpty which only reads from environment variables
    config = DecoupleConfig(RepositoryEmpty())

OPENROUTER_API_KEY = config.get("OPENROUTER_API_KEY", default="", cast=str)
OPENROUTER_BASE_URL = config.get("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1", cast=str)
OPENROUTER_MODEL = config.get("OPENROUTER_MODEL", default="qwen/qwen-plus", cast=str)
OPENROUTER_MAX_TOKENS = 12000  # Maximum allowed tokens for OpenRouter API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300  # Don't use the full max tokens to avoid hitting the limit

from openai import AsyncOpenAI

# Initialize Langfuse (optional)
LANGFUSE_PUBLIC_KEY = config.get("LANGFUSE_PUBLIC_KEY", default="", cast=str)
LANGFUSE_SECRET_KEY = config.get("LANGFUSE_SECRET_KEY", default="", cast=str)
LANGFUSE_HOST = config.get("LANGFUSE_HOST", default="http://langfuse:3000",
                           cast=str)  # Internal Docker network uses port 3000
ENABLE_LANGFUSE = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

langfuse_client = None
if ENABLE_LANGFUSE:
    try:
        from langfuse import Langfuse

        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logging.info("Langfuse client initialized for LLM observability.")
    except Exception as e:
        logging.warning(f"Failed to initialize Langfuse: {e}. Continuing without observability.")
        langfuse_client = None

# Wrap AsyncOpenAI with Langfuse if enabled
if ENABLE_LANGFUSE and langfuse_client:
    from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

    openrouter_client = LangfuseAsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    ) if OPENROUTER_API_KEY else None
else:
    openrouter_client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    ) if OPENROUTER_API_KEY else None

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def update_langfuse_generation_cost(response, operation_name: str = ""):
    """
    Extract cost from OpenRouter API response and update Langfuse generation.
    The Langfuse OpenAI wrapper should automatically capture cost, but we manually
    update it here to ensure it's properly set from OpenRouter's usage accounting.
    
    Args:
        response: OpenRouter API response object
        operation_name: Name of the operation for logging
    """
    if not response or not ENABLE_LANGFUSE or not langfuse_client:
        return
    
    try:
        # Extract usage information from OpenRouter response
        usage = getattr(response, 'usage', None)
        if not usage:
            return
        
        # Extract cost (in credits) - OpenRouter returns cost in the usage object
        cost = getattr(usage, 'cost', None)
        if cost is None:
            return
        
        prompt_tokens = getattr(usage, 'prompt_tokens', None)
        completion_tokens = getattr(usage, 'completion_tokens', None)
        total_tokens = getattr(usage, 'total_tokens', None)
        
        # Log cost information for debugging
        logging.debug("OpenRouter usage for %s: cost=%.6f credits | tokens: prompt=%s completion=%s total=%s",
                     operation_name or "operation", cost,
                     prompt_tokens or "N/A", completion_tokens or "N/A", total_tokens or "N/A")
        
        # Note: The Langfuse OpenAI wrapper should automatically extract cost from OpenRouter
        # responses when extra_body={"usage": {"include": True}} is used.
        # If cost is not appearing in Langfuse UI, check:
        # 1. That OpenRouter is returning cost in the usage object (check logs)
        # 2. That Langfuse is properly configured to recognize OpenRouter models
        # 3. That the Langfuse version supports OpenRouter cost extraction
        # 
        # The cost information is logged above for debugging purposes.
        # To manually set cost, you may need to use @observe() decorators or
        # manually create generations using langfuse_client.generation()
    except Exception as e:
        logging.debug(f"Failed to extract/update OpenRouter cost: {e}")


# API Interaction Functions
async def generate_completion(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    """Generate completion using OpenRouter API."""
    return await generate_completion_from_openrouter(prompt, max_tokens)


def get_tokenizer(model_name: str):
    # OpenRouter models typically use OpenAI-compatible tokenizers
    # Try to extract the base model name (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
    if "/" in model_name:
        base_model = model_name.split("/")[-1]
    else:
        base_model = model_name

    if base_model.lower().startswith("gpt-") or model_name.lower().startswith("openai/"):
        try:
            return tiktoken.encoding_for_model(base_model)
        except Exception:
            # Fallback to cl100k_base for GPT models
            return tiktoken.get_encoding("cl100k_base")
    elif model_name.lower().startswith("claude-") or model_name.lower().startswith("anthropic/"):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    elif model_name.lower().startswith("llama-") or model_name.lower().startswith("meta/"):
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b", clean_up_tokenization_spaces=False)
    else:
        # Default to cl100k_base for unknown models (OpenRouter often uses OpenAI-compatible models)
        logging.warning(f"Unknown model format: {model_name}, using cl100k_base tokenizer")
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
        return approximate_tokens(text)


def approximate_tokens(text: str) -> int:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and punctuation, keeping punctuation
    tokens = re.findall(r'\b\w+\b|\S', text)
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
        elif re.match(r'^[A-Z]{2,}$', token):  # Acronyms
            count += len(token)
        elif re.search(r'[^\w\s]', token):  # Punctuation and special characters
            count += 1
        elif len(token) > 10:  # Long words often split into multiple tokens
            count += len(token) // 4 + 1
        else:
            count += 1
    # Add a 10% buffer for potential underestimation
    return int(count * 1.1)


def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    chunks = []
    tokenizer = get_tokenizer(model_name)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    adjusted_chunks = adjust_overlaps(chunks, tokenizer, max_chunk_tokens)
    return adjusted_chunks


def split_long_sentence(sentence: str, max_tokens: int, model_name: str) -> List[str]:
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    tokenizer = get_tokenizer(model_name)

    for word in words:
        word_tokens = len(tokenizer.encode(word))
        if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def adjust_overlaps(chunks: List[str], tokenizer, max_chunk_tokens: int, overlap_size: int = 50) -> List[str]:
    adjusted_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            adjusted_chunks.append(chunks[i])
        else:
            overlap_tokens = len(tokenizer.encode(' '.join(chunks[i - 1].split()[-overlap_size:])))
            current_tokens = len(tokenizer.encode(chunks[i]))
            if overlap_tokens + current_tokens > max_chunk_tokens:
                overlap_adjusted = chunks[i].split()[:-overlap_size]
                adjusted_chunks.append(' '.join(overlap_adjusted))
            else:
                adjusted_chunks.append(' '.join(chunks[i - 1].split()[-overlap_size:] + chunks[i].split()))

    return adjusted_chunks


async def generate_completion_from_openrouter(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    if not openrouter_client:
        logging.error("OpenRouter client not initialized. Please set OPENROUTER_API_KEY environment variable.")
        return None
    if not OPENROUTER_API_KEY:
        logging.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        return None

    # Create Langfuse trace if enabled
    trace = None
    if ENABLE_LANGFUSE and langfuse_client:
        trace = langfuse_client.trace(
            name="llm_aided_ocr_completion",
            metadata={
                "model": OPENROUTER_MODEL,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
            }
        )

    prompt_tokens = estimate_tokens(prompt, OPENROUTER_MODEL)
    adjusted_max_tokens = min(max_tokens, OPENROUTER_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for OpenRouter API. Chunking the input.")
        chunks = chunk_text(prompt, OPENROUTER_MAX_TOKENS - TOKEN_CUSHION, OPENROUTER_MODEL)
        results = []
        for i, chunk in enumerate(chunks):
            try:
                # Create span for each chunk
                span = None
                if trace:
                    span = trace.span(
                        name=f"ocr_chunk_{i + 1}",
                        metadata={"chunk_index": i, "chunk_length": len(chunk)},
                    )

                response = await openrouter_client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[{"role": "user", "content": chunk}],
                    max_tokens=min(adjusted_max_tokens, OPENROUTER_MAX_TOKENS // 2),
                    temperature=0.7,
                    extra_body={"usage": {"include": True}},  # Enable OpenRouter usage accounting
                    name=f"ocr_chunk_{i + 1}",  # Name for Langfuse generation tracking
                )
                result = response.choices[0].message.content
                results.append(result)

                # Update Langfuse generation with cost information from OpenRouter
                update_langfuse_generation_cost(response, f"ocr_chunk_{i + 1}")

                if span:
                    span.update(output={"result_length": len(result) if result else 0})

                if hasattr(response, 'usage') and response.usage:
                    logging.info(f"Chunk processed. Output tokens: {response.usage.completion_tokens:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk with OpenRouter: {e}")
                if trace:
                    trace.update(metadata={"error": str(e)})
        final_result = " ".join(results)
        if trace:
            trace.update(output={"final_result_length": len(final_result)})
        return final_result
    else:
        try:
            response = await openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
                extra_body={"usage": {"include": True}},  # Enable OpenRouter usage accounting
                name="llm_aided_ocr_completion",  # Name for Langfuse generation tracking
            )
            output_text = response.choices[0].message.content
            if hasattr(response, 'usage') and response.usage:
                logging.info(f"Total tokens: {response.usage.total_tokens:,}")
            logging.info(f"Generated output (abbreviated): {output_text[:150]}...")

            # Update Langfuse generation with cost information from OpenRouter
            update_langfuse_generation_cost(response, "llm_aided_ocr_completion")

            if trace:
                trace.update(
                    output={"output_length": len(output_text) if output_text else 0,
                            "output_preview": output_text[:200] if output_text else ""},
                )

            return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from OpenRouter API: {e}")
            if trace:
                trace.update(metadata={"error": str(e)})
            return None


# Image Processing Functions
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    return Image.fromarray(gray)


def convert_pdf_to_images(input_pdf_file_path: str, max_pages: int = 0, skip_first_n_pages: int = 0) -> List[
    Image.Image]:
    logging.info(f"Processing PDF file {input_pdf_file_path}")
    if max_pages == 0:
        last_page = None
        logging.info("Converting all pages to images...")
    else:
        last_page = skip_first_n_pages + max_pages
        logging.info(f"Converting pages {skip_first_n_pages + 1} to {last_page}")
    first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing
    images = convert_from_path(input_pdf_file_path, first_page=first_page, last_page=last_page)
    logging.info(f"Converted {len(images)} pages from PDF file to images.")
    return images


def ocr_image(image, lang_code: str = "eng"):
    """
    Perform OCR on an image using Tesseract with specified language.
    lang_code: Tesseract language code (e.g., 'eng', 'vie', 'chi_sim')
    """
    preprocessed_image = preprocess_image(image)
    try:
        return pytesseract.image_to_string(preprocessed_image, lang=lang_code)
    except Exception as e:
        logging.warning(f"OCR failed with lang={lang_code}, falling back to eng: {e}")
        # Fallback to English if language pack not available
        return pytesseract.image_to_string(preprocessed_image, lang="eng")


async def process_chunk(chunk: str, prev_context: str, chunk_index: int, total_chunks: int, reformat_as_markdown: bool,
                        suppress_headers_and_page_numbers: bool) -> Tuple[str, str]:
    logging.info(f"Processing chunk {chunk_index + 1}/{total_chunks} (length: {len(chunk):,} characters)")

    # Step 1: OCR Correction
    ocr_correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the original text
   - Do not add any new information not present in the original text
   - Remove unnecessary line breaks within sentences or paragraphs
   - Maintain paragraph breaks
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.

Previous context:
{prev_context[-500:]}

Current chunk to process:
{chunk}

Corrected text:
"""

    ocr_corrected_chunk = await generate_completion(ocr_correction_prompt, max_tokens=len(chunk) + 500)

    processed_chunk = ocr_corrected_chunk

    # Step 2: Markdown Formatting (if requested)
    if reformat_as_markdown:
        markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
   - Ensure each heading is on its own line
   - Add a blank line before and after each heading
2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for example, "cor- rect" should be "correct")
3. Format lists properly (unordered or ordered) if they exist in the original text
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
5. Preserve all original content and meaning
6. Do not add any extra punctuation or modify the existing punctuation
7. Remove any spuriously inserted introductory text such as "Here is the corrected text:" that may have been added by the LLM and which is obviously not part of the original text.
8. Remove any obviously duplicated content that appears to have been accidentally included twice. Follow these strict guidelines:
   - Remove only exact or near-exact repeated paragraphs or sections within the main chunk.
   - Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries.
   - Do not remove content that is simply similar but conveys different information.
   - Preserve all unique content, even if it seems redundant.
   - Ensure the text flows smoothly after removal.
   - Do not add any new content or explanations.
   - If no obvious duplicates are found, return the main chunk unchanged.
9. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as blockquotes." if not suppress_headers_and_page_numbers else "Carefully remove headers, footers, and page numbers while preserving all other content."}

Text to reformat:

{ocr_corrected_chunk}

Reformatted markdown:
"""
        processed_chunk = await generate_completion(markdown_prompt, max_tokens=len(ocr_corrected_chunk) + 500)
    new_context = processed_chunk[-1000:]  # Use the last 1000 characters as context for the next chunk
    logging.info(
        f"Chunk {chunk_index + 1}/{total_chunks} processed. Output length: {len(processed_chunk):,} characters")
    return processed_chunk, new_context


async def process_chunks(chunks: List[str], reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> \
List[str]:
    """Process chunks concurrently using OpenRouter API."""
    total_chunks = len(chunks)

    async def process_chunk_with_context(chunk: str, prev_context: str, index: int) -> Tuple[int, str, str]:
        processed_chunk, new_context = await process_chunk(chunk, prev_context, index, total_chunks,
                                                           reformat_as_markdown, suppress_headers_and_page_numbers)
        return index, processed_chunk, new_context

    logging.info("Processing chunks concurrently with OpenRouter API...")
    tasks = [process_chunk_with_context(chunk, "", i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    # Sort results by index to maintain order
    sorted_results = sorted(results, key=lambda x: x[0])
    processed_chunks = [chunk for _, chunk, _ in sorted_results]
    logging.info(f"All {total_chunks} chunks processed successfully")
    return processed_chunks


async def process_document(list_of_extracted_text_strings: List[str], reformat_as_markdown: bool = True,
                           suppress_headers_and_page_numbers: bool = True) -> str:
    logging.info(f"Starting document processing. Total pages: {len(list_of_extracted_text_strings):,}")
    full_text = "\n\n".join(list_of_extracted_text_strings)
    logging.info(f"Size of full text before processing: {len(full_text):,} characters")
    chunk_size, overlap = 8000, 10
    # Improved chunking logic
    paragraphs = re.split(r'\n\s*\n', full_text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_chunk_length + paragraph_length <= chunk_size:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length
        else:
            # If adding the whole paragraph exceeds the chunk size,
            # we need to split the paragraph into sentences
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = []
            current_chunk_length = 0
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_chunk_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_chunk_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_length = sentence_length
    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
    # Add overlap between chunks
    for i in range(1, len(chunks)):
        overlap_text = chunks[i - 1].split()[-overlap:]
        chunks[i] = " ".join(overlap_text) + " " + chunks[i]
    logging.info(f"Document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
    processed_chunks = await process_chunks(chunks, reformat_as_markdown, suppress_headers_and_page_numbers)
    final_text = "".join(processed_chunks)
    logging.info(f"Size of text after combining chunks: {len(final_text):,} characters")
    logging.info(f"Document processing complete. Final text length: {len(final_text):,} characters")
    return final_text


def remove_corrected_text_header(text):
    return text.replace("# Corrected text\n", "").replace("# Corrected text:", "").replace("\nCorrected text",
                                                                                           "").replace(
        "Corrected text:", "")


async def assess_output_quality(original_text, processed_text):
    max_chars = 15000  # Limit to avoid exceeding token limits
    available_chars_per_text = max_chars // 2  # Split equally between original and processed

    original_sample = original_text[:available_chars_per_text]
    processed_sample = processed_text[:available_chars_per_text]

    prompt = f"""Compare the following samples of original OCR text with the processed output and assess the quality of the processing. Consider the following factors:
1. Accuracy of error correction
2. Improvement in readability
3. Preservation of original content and meaning
4. Appropriate use of markdown formatting (if applicable)
5. Removal of hallucinations or irrelevant content

Original text sample:
```
{original_sample}
```

Processed text sample:
```
{processed_sample}
```

Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.

Your response should be in the following format:
SCORE: [Your score]
EXPLANATION: [Your explanation]
"""

    response = await generate_completion(prompt, max_tokens=1000)

    try:
        lines = response.strip().split('\n')
        score_line = next(line for line in lines if line.startswith('SCORE:'))
        score = int(score_line.split(':')[1].strip())
        explanation = '\n'.join(line for line in lines if line.startswith('EXPLANATION:')).replace('EXPLANATION:',
                                                                                                   '').strip()
        logging.info(f"Quality assessment: Score {score}/100")
        logging.info(f"Explanation: {explanation}")
        return score, explanation
    except Exception as e:
        logging.error(f"Error parsing quality assessment response: {e}")
        logging.error(f"Raw response: {response}")
        return None, None


async def main():
    try:
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        input_pdf_file_path = '160301289-Warren-Buffett-Katharine-Graham-Letter.pdf'
        max_test_pages = 0
        skip_first_n_pages = 0
        reformat_as_markdown = True
        suppress_headers_and_page_numbers = True

        logging.info(f"Using OpenRouter API for completions")
        logging.info(f"Using OpenRouter model: {OPENROUTER_MODEL}")

        base_name = os.path.splitext(input_pdf_file_path)[0]
        output_extension = '.md' if reformat_as_markdown else '.txt'

        raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
        llm_corrected_output_file_path = base_name + '_llm_corrected' + output_extension

        list_of_scanned_images = convert_pdf_to_images(input_pdf_file_path, max_test_pages, skip_first_n_pages)
        logging.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        logging.info("Extracting text from converted pages...")
        with ThreadPoolExecutor() as executor:
            list_of_extracted_text_strings = list(executor.map(ocr_image, list_of_scanned_images))
        logging.info("Done extracting text from converted pages.")
        raw_ocr_output = "\n".join(list_of_extracted_text_strings)
        with open(raw_ocr_output_file_path, "w") as f:
            f.write(raw_ocr_output)
        logging.info(f"Raw OCR output written to: {raw_ocr_output_file_path}")

        logging.info("Processing document...")
        final_text = await process_document(list_of_extracted_text_strings, reformat_as_markdown,
                                            suppress_headers_and_page_numbers)
        cleaned_text = remove_corrected_text_header(final_text)

        # Save the LLM corrected output
        with open(llm_corrected_output_file_path, 'w') as f:
            f.write(cleaned_text)
        logging.info(f"LLM Corrected text written to: {llm_corrected_output_file_path}")

        if final_text:
            logging.info(f"First 500 characters of LLM corrected processed text:\n{final_text[:500]}...")
        else:
            logging.warning("final_text is empty or not defined.")

        logging.info(f"Done processing {input_pdf_file_path}.")
        logging.info("\nSee output files:")
        logging.info(f" Raw OCR: {raw_ocr_output_file_path}")
        logging.info(f" LLM Corrected: {llm_corrected_output_file_path}")

        # Perform a final quality check
        quality_score, explanation = await assess_output_quality(raw_ocr_output, final_text)
        if quality_score is not None:
            logging.info(f"Final quality score: {quality_score}/100")
            logging.info(f"Explanation: {explanation}")
        else:
            logging.warning("Unable to determine final quality score.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    asyncio.run(main())
