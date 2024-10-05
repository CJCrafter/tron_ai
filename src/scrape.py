import openai
import requests
import markdownify
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from src.settings import SETTINGS


openai.api_key = SETTINGS.openai_api_key

pinecone_api_key = SETTINGS.pinecone_api_key
pc = Pinecone(api_key=pinecone_api_key)
index_name = "tron"
embedding_dimension = 3072
index = pc.Index(index_name)


# List of URLs to process
urls = [
    "https://developers.tron.network/docs/getting-start",
    "https://developers.tron.network/docs/build-a-web3-app",
    "https://developers.tron.network/docs/account",
    "https://developers.tron.network/docs/resource-model",
    "https://developers.tron.network/docs/staking-on-tron-network",
    "https://developers.tron.network/docs/super-representatives",
    "https://developers.tron.network/docs/becoming-a-super-representative",
    "https://developers.tron.network/docs/tron-protocol-transaction",
    "https://developers.tron.network/docs/block",
    "https://developers.tron.network/docs/tvm",
    "https://developers.tron.network/docs/event",
    "https://developers.tron.network/docs/vm-exception-handling",
    "https://developers.tron.network/docs/nodes-and-clients",
    "https://developers.tron.network/docs/networks",
    "https://developers.tron.network/docs/multi-signature",
    "https://developers.tron.network/docs/multi-signature-example-process-flow",
    "https://developers.tron.network/docs/concensus",
    "https://developers.tron.network/docs/token-standards-overview",
    "https://developers.tron.network/docs/token-standards-trx",
    "https://developers.tron.network/docs/trc10",
    "https://developers.tron.network/docs/trc10-transfer-in-smart-contracts",
    "https://developers.tron.network/docs/trc20-protocol-interface",
    "https://developers.tron.network/docs/issuing-trc20-tokens-tutorial",
    "https://developers.tron.network/docs/trc20-contract-interaction",
    "https://developers.tron.network/docs/get-trc20-transaction-history",
    "https://developers.tron.network/docs/trc-721",
    "https://developers.tron.network/docs/trc-721-protocol-interface",
    "https://developers.tron.network/docs/trc-721-contract-example",
    "https://developers.tron.network/docs/trc-721-token-issuance",
    "https://developers.tron.network/docs/trc-721-contract-interaction",
    "https://developers.tron.network/docs/uploading-nft-metadata-to-btfs-network",
    "https://developers.tron.network/docs/smart-contracts-introduction",
    "https://developers.tron.network/docs/smart-contract-language",
    "https://developers.tron.network/docs/stake-20-solidity-api",
    "https://developers.tron.network/docs/smart-contract-deployment-and-invocation",
    "https://developers.tron.network/docs/parameter-encoding-and-decoding",
    "https://developers.tron.network/docs/set-feelimit",
    "https://developers.tron.network/docs/smart-contract-security",
    "https://developers.tron.network/docs/tronz-implementation-details",
    "https://developers.tron.network/docs/how-to-use-shielded-smart-contracts",
    "https://developers.tron.network/docs/deploy-the-fullnode-or-supernode",
    "https://developers.tron.network/docs/main-net-database-snapshots",
    "https://developers.tron.network/docs/tron-private-chain",
    "https://developers.tron.network/docs/event-subscription",
    "https://developers.tron.network/docs/event-plugin-deployment-mongodb",
    "https://developers.tron.network/docs/event-plugin-deployment-kafka",
    "https://developers.tron.network/docs/use-java-trons-built-in-message-queue-for-event-subscription",
    "https://developers.tron.network/docs/dapp-development-tools",
    "https://developers.tron.network/docs/trongrid",
    "https://developers.tron.network/docs/tron-ide",
    "https://developers.tron.network/docs/walletconnect-tron",
    "https://developers.tron.network/docs/tronwallet-adapter",
    "https://developers.tron.network/docs/tronwidgets",
    "https://developers.tron.network/docs/smart-contract-development",
    "https://developers.tron.network/docs/creating-and-compiling",
    "https://developers.tron.network/docs/deploying",
    "https://developers.tron.network/docs/querying-the-contract-data",
    "https://developers.tron.network/docs/smart-contract-interaction",
    "https://developers.tron.network/docs/tronlink-integration",
    "https://developers.tron.network/docs/tronlink-events",
    "https://developers.tron.network/docs/adding-assets-to-tronlink",
    "https://developers.tron.network/docs/exchangewallet-integrate-with-the-tron-network",
    "https://developers.tron.network/docs/api-signature-and-broadcast-flow",
    "https://developers.tron.network/docs/bittorrent-chain",
    "https://developers.tron.network/docs/band-oracle",
    "https://developers.tron.network/docs/rosetta-api",
    "https://developers.tron.network/docs/glossary",
    "https://developers.tron.network/docs/faq",
    "https://developers.tron.network/docs/announcements",
]

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=SETTINGS.openai_api_key, model="text-embedding-3-large")

# Prepare the headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

# Initialize a list to collect all documents
all_documents = []

for url in urls:
    print("Processing URL:", url)
    try:
        # Fetch the HTML content from the website
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        html_content = response.text

        # Parse HTML and extract the div with id 'content-container'
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', id='content-container')
        if not content_div:
            print(f"No content-container found at {url}")
            continue  # Skip this URL if the div is not found

        # Convert the content div to markdown
        markdown_content = markdownify.markdownify(str(content_div), heading_style="ATX")

        # Use MarkdownHeaderTextSplitter to split the markdown content
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        documents = markdown_splitter.split_text(markdown_content)

        # Add URL metadata to each document
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = url

        all_documents.extend(documents)

    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")

if not all_documents:
    print("No documents to process.")
    exit()


# Further split the documents if needed using RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
split_documents = text_splitter.split_documents(all_documents)

# Generate embeddings and prepare vectors
vectors = []
for i, doc in enumerate(split_documents):
    print(f"Processing document {i} of {len(split_documents)}")
    try:
        # Generate the embedding for the document
        embedding = embeddings.embed_query(doc.page_content)

        # Prepare metadata including headers and source URL
        metadata = doc.metadata
        metadata['text'] = doc.page_content

        # Prepare the vector
        vector = {
            'id': f'doc-{i}',
            'values': embedding,
            'metadata': metadata
        }
        vectors.append(vector)

    except Exception as e:
        print(f"Failed to process document {i}: {e}")

# Upload the vectors to Pinecone in batches
batch_size = 100  # Adjust the batch size as needed
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)

print("All vectors have been uploaded to Pinecone.")
