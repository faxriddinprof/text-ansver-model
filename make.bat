@echo off

IF "%1"=="" (
    echo Usage: make.bat [command]
    echo.
    echo Commands:
    echo   run            - Start Django development server
    echo   migrate        - Run database migrations
    echo   makemigrations - Create new migrations
    echo   install        - Install dependencies
    echo   shell          - Open Django shell
    echo   test           - Run test.py evaluation
    GOTO end
)


IF "%1"=="install" (
    pip install -r requirements.txt
    GOTO end
)


IF "%1"=="test" (
    python test.py
    GOTO end
)

echo Unknown command: %1
echo Run "make.bat" without arguments to see available commands.

:end
