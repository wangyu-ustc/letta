from typing import List, Optional

from letta.agent import Agent

def raise_error(self: "Agent", error_type: str) -> Optional[str]:
    """
    Sends an error message to the human user.

    Args:
        error_type (str): Type of error to raise, it needs to be from the following list:
        - "missing_images": When the user provided images in the text description but did not include the images in the content. 
    
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None

def send_message(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    return None


def conversation_search(self: "Agent", query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """

    import math

    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    from letta.utils import json_dumps

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
    # TODO: add paging by page number. currently cursor only works with strings.
    # original: start=page * count
    messages = self.message_manager.list_user_messages_for_agent(
        agent_id=self.agent_state.id,
        actor=self.user,
        query_text=query,
        limit=count,
    )
    total = len(messages)
    num_pages = math.ceil(total / count) - 1  # 0 index
    if len(messages) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(messages)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [message.text for message in messages]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str

def read_image(self: "Agent", image_urls: List[str]) -> Optional[str]:
    """
    Read the contents of an image.

    Args:
        image_urls (array[str]): List of the URLs of the images to read.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    import base64
    
    image_dir = self.image_dir if hasattr(self, "image_dir") and self.image_dir is not None else "."

    if isinstance(image_urls, str) and "," in image_urls:
        image_urls = image_urls.split(",")
    elif isinstance(image_urls, str):
        image_urls = [image_urls]

    image_messages = []

    for image_url in image_urls:

        with open(f"{image_dir}/{image_url}", "rb") as image_file:

            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_messages.append({
                'type': "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })

    return image_messages

def archival_memory_insert(self: "Agent", content: str, image_indices: str = None, descriptions: str=None) -> Optional[str]:
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): The content and rough idea in this archival memory. Try to be distinguishable as this is used for future retrieval. All unicode (including emojis) are supported.
        image_indices (str): Images that are associated with this piece of memory, e.g., '1,2,3,...,n'. Defaults to None.
        descriptions (str): Descriptions of of all the images in one string, e.g., "'1:description_1, 2:description_2, 3:description_3, ..., n:description_n]". Try to be as specific as possible, do not include general information. Defaults to None.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    if image_indices is not None and descriptions is not None:
        self.passage_manager.insert_passage(
            agent_state=self.agent_state,
            agent_id=self.agent_state.id,
            text=content + descriptions,
            image_url=image_indices,
            actor=self.user,
        )

    else:
        self.passage_manager.insert_passage(
            agent_state=self.agent_state,
            agent_id=self.agent_state.id,
            text=content,
            actor=self.user,
        )
    return None


def archival_memory_search(self: "Agent", query: str, page: Optional[int] = 0, start: Optional[int] = 0) -> Optional[str]:
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).
        start (Optional[int]): Starting index for the search results. Defaults to 0.

    Returns:
        str: Query result string
    """

    from letta.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except:
        raise ValueError(f"'page' argument must be an integer")
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

    try:
        # Get results using passage manager
        all_results = self.agent_manager.list_passages(
            actor=self.user,
            agent_id=self.agent_state.id,
            query_text=query,
            limit=count + start,  # Request enough results to handle offset
            embedding_config=self.agent_state.embedding_config,
            embed_query=True,
        )

        # Apply pagination
        end = min(count + start, len(all_results))
        paged_results = all_results[start:end]

        # Format results to match previous implementation
        formatted_results = [{"timestamp": str(result.created_at), "content": result.text, "image_url": result.image_url} for result in paged_results]

        return formatted_results, len(formatted_results)

    except Exception as e:
        raise e


def core_memory_append(agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
    """
    Append to the contents of core memory.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    new_value = current_value + "\n" + str(content)
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None


def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> Optional[str]:  # type: ignore
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    if old_content not in current_value:
        raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    new_value = current_value.replace(str(old_content), str(new_content))
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None
