#!/bin/bash
cd $(dirname "$0")
CC=g++
OS=`uname`
FLAGS="-std=c++11"
if command -v clang-format >/dev/null 2>&1; then
  echo "Linting..."
  clang-format -i *.cpp *.hpp
fi
echo "Compiling..."
if [ $OS == "Darwin" ]; then
  CC=g++-6
fi
echo "... using $CC."
rm *.out
$CC $FLAGS -c $(find . -name \*.cpp) -lm
if [ $OS == "Darwin" ]; then
  rm benchmark-native.o
fi
echo "Building..."

$CC $FLAGS $(find . -name \*.o -not -name scggm.o) -o test.out -lm
$CC $FLAGS $(find . -name \*.o -not -name test.o) -o scggm.out -lm
rm $(find . -name \*.o)
