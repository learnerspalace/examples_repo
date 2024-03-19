from typing import Any, Dict, List

from pydantic import BaseModel, root_validator
import pdfminer

class GPT4AllEmbeddings(BaseModel):
    """GPT4All embedding models.

    To use, you should have the gpt4all python package installed

    Example:
        .. code-block:: python

            from langchain_community.embeddings import GPT4AllEmbeddings

            embeddings = GPT4AllEmbeddings()
    """

    # mydata: Any  #: :meta private:

    @root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed."""

        values["mydata"] = "Embed4All()"
        return values

    def embed_documents(
        self,
    ) -> List[List[float]]:
        """Embed a list of documents using GPT4All.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return self.mydata

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using GPT4All.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


embeddings = GPT4AllEmbeddings()
print(embeddings.embed_documents())
