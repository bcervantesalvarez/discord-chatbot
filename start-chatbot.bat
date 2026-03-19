@echo off
REM start-chatbot.bat -- Launch Discord Chatbot watchdog minimized.
REM Drop a shortcut to this file in:
REM   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
REM to auto-start Discord Chatbot when Windows boots.

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -WindowStyle Minimized -File "%~dp0start-chatbot.ps1"
