Sub copyline()
'
' This Macro can copy files line by line
' Please specify outputFileName to the output path
'
Dim outputFileName, outputFile
outputFileName = "/Users/jun/Dropbox/Documents/USC/18F/NIH/docs/h.txt"
outputFile = FreeFile
Open outputFileName For Output As #outputFile
Selection.MoveUp Unit:=wdLine, Count:=1000
Dim counter
Do
    Selection.Expand wdLine
    Print #outputFile, Selection.Text
    Selection.Collapse Direction:=wdCollapseStart
    Selection.MoveDown Unit:=wdLine, Count:=1
    counter = counter + 1
Loop Until counter = 774
Close #outputFile
End Sub