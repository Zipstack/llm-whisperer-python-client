# LLMWhisperer Python Client

LLMs are powerful, but their output is as good as the input you provide. LLMWhisperer is a technology that presents data from complex documents (different designs and formats) to LLMs in a way that they can best understand. LLMWhisperer features include Layout Preserving Mode, Auto-switching between native text and OCR modes, proper representation of radio buttons and checkboxes in PDF forms as raw text, among other features. You can now extract raw text from complex PDF documents or images without having to worry about whether the document is a native text document, a scanned image or just a picture clicked on a smartphone. Extraction of raw text from invoices, purchase orders, bank statements, etc works easily for structured data extraction with LLMs powered by LLMWhisperer's Layout Preserving mode. 

Refer to the client documentation for more information: [LLMWhisperer Client Documentation](https://docs.unstract.com/llm_whisperer/python_client/llm_whisperer_python_client_intro)

## Features

- Easy to use Pythonic interface.
- Handles all the HTTP requests and responses for you.
- Raises Python exceptions for API errors.

## Installation

You can install the LLMWhisperer Python Client using pip:

```bash
pip install llmwhisperer-client
```

## Usage

First, import the `LLMWhispererClient` from the `client` module:

```python
from unstract.llmwhisperer.client import LLMWhispererClient
```

Then, create an instance of the `LLMWhispererClient`:

```python
client = LLMWhispererClient(base_url="https://llmwhisperer-api.unstract.com/v1", api_key="your_api_key")
```

Now, you can use the client to interact with the LLMWhisperer API:

```python
# Get usage info
usage_info = client.get_usage_info()

# Process a document
# Extracted text is available in the 'extracted_text' field of the result
whisper = client.whisper(file_path="path_to_your_file")

# Get the status of a whisper operation
# whisper_hash is available in the 'whisper_hash' field of the result of the whisper operation
status = client.whisper_status(whisper_hash)

# Retrieve the result of a whisper operation
# whisper_hash is available in the 'whisper_hash' field of the result of the whisper operation
whisper = client.whisper_retrieve(whisper_hash)
```

### Error Handling

The client raises `LLMWhispererClientException` for API errors:

```python
try:
    result = client.whisper_retrieve("invalid_hash")
except LLMWhispererClientException as e:
    print(f"Error: {e.message}, Status Code: {e.status_code}")
```

### Simple use case with defaults

```python
client = LLMWhispererClient()
try:
    result = client.whisper(file_path="sample_files/restaurant_invoice_photo.pdf")
    extracted_text = result["extracted_text"]
    print(extracted_text)
except LLMWhispererClientException as e:
    print(e)
```

### Simple use case with more options set
We are forcing text processing and extracting text from the first two pages only.

```python
client = LLMWhispererClient()
try:
    result = client.whisper(
        file_path="sample_files/credit_card.pdf",
        processing_mode="text",
        force_text_processing=True,
        pages_to_extract="1,2",
    )
    extracted_text = result["extracted_text"]
    print(extracted_text)
except LLMWhispererClientException as e:
    print(e)
```

### Extraction with timeout set 

The platform has a hard timeout of 200 seconds. If the document takes more than 200 seconds to convert (large documents), the platform will switch to async extraction and return a hash. The client can be used to check the status of the extraction and retrieve the result. Also note that the timeout is in seconds and can be set by the caller too.


```python
client = LLMWhispererClient()
try:
    result = client.whisper(
        file_path="sample_files/credit_card.pdf",
        pages_to_extract="1,2",
        timeout=2,
    )
    if result["status_code"] == 202:
        print("Timeout occured. Whisper request accepted.")
        print(f"Whisper hash: {result['whisper-hash']}")
        while True:
            print("Polling for whisper status...")
            status = client.whisper_status(whisper_hash=result["whisper-hash"])
            if status["status"] == "processing":
                print("STATUS: processing...")
            elif status["status"] == "delivered":
                print("STATUS: Already delivered!")
                break
            elif status["status"] == "unknown":
                print("STATUS: unknown...")
                break
            elif status["status"] == "processed":
                print("STATUS: processed!")
                print("Let's retrieve the result of the extraction...")
                resultx = client.whisper_retrieve(
                    whisper_hash=result["whisper-hash"]
                )
                print(resultx["extracted_text"])
                break
            time.sleep(2)
except LLMWhispererClientException as e:
    print(e)
```

## Questions and Feedback

On Slack, [join great conversations](https://join-slack.unstract.com/) around LLMs, their ecosystem and leveraging them to automate the previously unautomatable!

[LLMWhisperer Playground](https://pg.llmwhisperer.unstract.com/): Test drive LLMWhisperer with your own documents. No sign up needed!

[LLMWhisperer developer documentation and playground](https://dev-pg.llmwhisperer.unstract.com/): Learn more about LLMWhisperer and its API.
