import language_tool_python

def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    if not matches:
        return "No grammar issues detected."
    
    issues = []
    for match in matches[:5]:  # limit to top 5
        issues.append(f"• {match.context} → {match.message}")
    
    return "\n".join(issues)
