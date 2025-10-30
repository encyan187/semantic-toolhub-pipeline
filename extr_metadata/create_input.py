
def create_input(tool_metadata):

    if tool_metadata is None:
        print("Dictionary is None")

    if isinstance(tool_metadata, dict) and len(tool_metadata) == 1:
        inner = next(iter(tool_metadata.values()))
        if isinstance(inner, dict):
            tool_metadata = inner

    parts = []

    if 'keywords' in tool_metadata:
        kw = tool_metadata['keywords']
        if kw:
            if isinstance(kw, list):
                text = ', '.join(str(x) for x in kw)
            else:
                text = str(kw)
            text = text.strip()
            if text:
                parts.append(text)

    # 2. Readme
    #readme = tool_metadata.get('readme', '')
    #if readme and readme.strip():
        #parts.append(readme.strip())

    # 3. Description
    desc = tool_metadata.get('wiki_description', '')
    if desc and desc.strip():
        parts.append(desc.strip())

    # Join with two newlines
    return ' '.join(parts)

