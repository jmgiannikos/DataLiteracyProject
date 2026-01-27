import re

def remove_math(tex_string, verbose=False): # remove math environments
    doclen = len(tex_string)
    if verbose:
        print(doclen)
    tex_string = re.sub(r"\\begin\{equation\}(?s:.)*?\\end\{equation\}", "", tex_string)
    if verbose:
        print(f"'equation' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(r"\$(?s:.)*?\$", "", tex_string)
    if verbose:
        print(f"'$' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"begin{math}") + r"(?s:.)*?" + re.escape(r"\end{math}"), "", tex_string) 
    if verbose:
        print(f"'math' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\(") + r"(?s:.)*?" + re.escape(r"\)"), "", tex_string) 
    if verbose:    
        print(f"'\(' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\[") + r"(?s:.)*?" + re.escape(r"\]"), "", tex_string) 
    if verbose:    
        print(f"'\[' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"begin{displaymath}") + r"(?s:.)*?" + re.escape(r"\end{displaymath}"), "", tex_string) 
    if verbose:    
        print(f"'displaymath' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"begin{equation*}") + r"(?s:.)*?" + re.escape(r"\end{equation*}"), "", tex_string)
    if verbose:
        print(f"'equation*' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    return tex_string

def remove_citations(tex_string, verbose=False): # remove citations
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\cite{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:
        print(f"'cite' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\citet{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:
        print(f"'citet' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\citep{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:    
        print(f"'citep' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    return tex_string

def remove_references(tex_string, verbose=False):
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\ref{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:
        print(f"'ref' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\cref{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:
        print(f"'cref' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\href{") + r"(?s:.)*?}\{" + r"(?s:.)*?}", "", tex_string)
    if verbose:    
        print(f"'href' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\href{") + r"(?s:.)*?}", "", tex_string)
    if verbose:
        print(f"'href (only url)' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    tex_string = re.sub(r"\\href", "", tex_string)
    if verbose:    
        print(f"'href (only command)' removed: {doclen - len(tex_string)}")
    return tex_string

def remove_algorithm(tex_string, verbose=False):
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\begin{algorithm}") + r"(?s:.)*?" + re.escape(r"\end{algorithm}"), "", tex_string)
    if verbose:
        print(f"'algorithm' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    return tex_string

def remove_align(tex_string, verbose=False):
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\begin{align}") + r"(?s:.)*?" + re.escape(r"\end{align}"), "", tex_string)
    if verbose:
        print(f"'align' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    return tex_string

def remove_url(tex_string, verbose=False):
    doclen = len(tex_string)
    tex_string = re.sub(re.escape(r"\url{") + r"(?s:.)*?\}", "", tex_string)
    if verbose:
        print(f"'url' removed: {doclen - len(tex_string)}")
    doclen = len(tex_string)
    return tex_string

def tagswap(tex_string, env_name):
    tex_string = re.sub(re.escape(r"\begin{" + env_name + r"}"), f"/B{env_name.upper()}/", tex_string)
    tex_string = re.sub(re.escape(r"\end{" + env_name + r"}"), f"/E{env_name.upper()}/", tex_string)
    return tex_string

def swaptag(tex_string, env_name):
    tex_string = re.sub(f"/B{env_name.upper()}/", r"\begin{" + env_name + r"}", tex_string)
    tex_string = re.sub(f"/E{env_name.upper()}/", r"\\end{" + env_name + r"}", tex_string)
    return tex_string

def removeall_begin(tex_string, verbose=False):
    doclen = len(tex_string)

    tex_string = tagswap(tex_string, "itemize")
    tex_string = tagswap(tex_string, "enumerate")
    tex_string = tagswap(tex_string, "document")

    tex_string = re.sub(re.escape(r"\begin") + r"(?s:.)*?" + re.escape(r"\\end"), "", tex_string)

    tex_string = swaptag(tex_string, "itemize")
    tex_string = swaptag(tex_string, "enumerate")
    tex_string = tagswap(tex_string, "document")

    if verbose:
        print(f"'misc' removed: {doclen - len(tex_string)}")
    return tex_string


TEX_PARSING_RULES_LIST = [remove_math, remove_citations, remove_references, remove_algorithm, remove_align, remove_url, removeall_begin]
