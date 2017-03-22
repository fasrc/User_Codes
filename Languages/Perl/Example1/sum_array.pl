#!/usr/bin/env perl
#==========================================================================
# Program: sum_array.pl
# Purpose: Creates a random array and sums up its elements
#          Array domension is supplied by the user.
#
# RUN:     perl sum_array.pl
#==========================================================================
print "Program generates a random array and prints out its elements.\n";
print "Please, enter array dimension: \n";
$N = <STDIN>;
for ( $i = 0; $i <= $N; $i++ ){
    $random_number= rand(); 
    $darr[$i] = $random_number;
#    print $darr[$i], "\n";
}
$isum = 0;
for ( $j = 0; $j <= $N; $j++ ){
    $isum = $isum + $darr[$j];
}
print "Array dimension: ", $N, "\n";
print "Sum of array elements: ", $isum, "\n";
