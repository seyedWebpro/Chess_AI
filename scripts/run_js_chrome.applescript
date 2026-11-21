on run argv
    set jsCode to item 1 of argv
    set urlFilter to ""
    if (count of argv) > 1 then
        set urlFilter to item 2 of argv
    end if
    
    tell application "Google Chrome"
        if urlFilter is not "" then
            repeat with w in windows
                set idx to 1
                repeat with t in tabs of w
                    if (URL of t) contains urlFilter then
                        set active tab index of w to idx
                        set index of w to 1
                        return execute t javascript jsCode
                    end if
                    set idx to idx + 1
                end repeat
            end repeat
        end if
        tell active tab of front window
            return execute javascript jsCode
        end tell
    end tell
end run
