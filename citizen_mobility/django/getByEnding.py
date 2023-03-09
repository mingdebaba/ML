def getByEnding(inputText):
    """_summary_

    Args:
        inputText (string): _description_

    Returns:
        _string_: _description_
    """

    if inputText[-6:] == "......":
        except_last6 = inputText[:-6]
        outputText = except_last6
    else:
        outputText = ""
    return outputText


inputText = "abcdasdad...."
inputText2 = "abasdadasdascd......"
print(getByEnding(inputText2))
