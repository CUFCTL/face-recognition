@ECHO OFF
set PATH=%IMPULSEC_HOME%\bin;%IMPULSEC_GCC_HOME%\bin;%PATH%;
set IMPULSEC_HOME=C:/Impulse/CoDeveloper3
set IMPULSEC_GCC_HOME=%IMPULSEC_HOME%/MinGW
"C:\Impulse\CoDeveloper3\bin\make.exe" SHELL=C:/Impulse/CoDeveloper3/bin/sh.exe -f"C:\Documents and Settings\smithmc\Desktop\mneeley\old_impulse_c\_Makefile" build
echo 1 > "CoDeveloper.log.done"
