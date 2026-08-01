# Greeting helpers.
function greet
    if test -n "$argv"
        echo "hi $argv"
    end
end

function farewell
    echo bye
end
