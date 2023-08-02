default(timer,1);

ismersenne(x)=ispseudoprime(2^x-1)

print("apply comparison")
apply(ismersenne,primes(400))
parapply(ismersenne,primes(400))

print("select comparison")
select(ismersenne,primes(400))
parselect(ismersenne,primes(400))
