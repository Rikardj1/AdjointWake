@echo off
rem Compile-check the STAR-CCM+ macros in this folder against the 21.04.008 jars.
rem Compiling is a check only; STAR compiles the macro itself at run time.
rem Usage: compile_check.bat [Macro.java ...]   (default: every *.java in this folder)
rem Override the locations with the JDK25 and STARJARS environment variables.
setlocal enabledelayedexpansion
if "%JDK25%"=="" set "JDK25=C:\Program Files\Java\jdk-25.0.5.0.3"
if "%STARJARS%"=="" set "STARJARS=C:\Users\Rikard\StarCCM_API\21.04.008\java\platform"
if not exist "%JDK25%\bin\javac.exe" (echo javac not found in "%JDK25%\bin" & exit /b 2)
if not exist "%STARJARS%" (echo STAR jar folder "%STARJARS%" not found & exit /b 2)
pushd "%~dp0"
set "OUT=%TEMP%\starcheck"
if not exist "%OUT%" mkdir "%OUT%"
set "ARGS=%OUT%\javac_args.txt"
rem Class path: one "folder/*" wildcard per folder that holds jars (keeps the line short)
set "CP="
if exist "%STARJARS%\*.jar" set "CP=%STARJARS%\*"
for /d /r "%STARJARS%" %%d in (*) do (
  if exist "%%d\*.jar" set "CP=!CP!;%%d\*"
)
if "!CP:~0,1!"==";" set "CP=!CP:~1!"
set "CP=!CP:\=/!"
set "OUTF=%OUT:\=/%"
> "%ARGS%" echo -proc:none
>> "%ARGS%" echo -nowarn
>> "%ARGS%" echo -d
>> "%ARGS%" echo "%OUTF%"
rem javac does not expand folder/* class path wildcards inside an @argfile, so -cp goes on the command line
if "%~1"=="" (
  for %%f in (*.java) do >> "%ARGS%" echo "%%f"
) else (
  for %%f in (%*) do >> "%ARGS%" echo "%%~f"
)
"%JDK25%\bin\javac.exe" -cp "!CP!" @"%ARGS%"
set "RC=%ERRORLEVEL%"
popd
if "%RC%"=="0" (echo COMPILE_OK) else (echo COMPILE_FAILED %RC%)
exit /b %RC%
