#!/bin/bash ; 

final_result=0

for f in ex1.cpp ex2.cpp ex3.cpp ex4.cpp ex5.cpp
do
  rm -f a.out
  syclcc -O2 $f && ./a.out > /dev/null  # ex4's barrier needs -O2 or -O3
  result=$?
  final_result=$((final_result || result))
done

case $final_result in
  0) echo "All tests passed ok." ;;
  1) echo "At least one test failed." ;;
  *) echo "error." ;;
esac

exit $final_result
