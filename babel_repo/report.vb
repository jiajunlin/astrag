' Reporting module.
Public Module Reports
    ' Sum of all order totals.
    Public Function GrandTotal(ByVal totals As List(Of Integer)) As Integer
        Dim s As Integer = 0
        For Each t In totals
            s += t
        Next
        Return s
    End Function
End Module
