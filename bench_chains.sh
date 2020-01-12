set +e

for CHAINS in 100 1000 10000
do
	hyperfine 'python jax_sampler.py --samples 1000 --chains '"$CHAINS"''
        hyperfine 'python tf_sampler.py --samples 1000 --chains '"$CHAINS"' --runonce True'
        hyperfine 'python tfp_sampler.py --samples 1000 --chains '"$CHAINS"' --runonce True'
        hyperfine 'python tf_sampler.py --samples 1000 --chains '"$CHAINS"' --runonce True --keep_all_samples False'
        hyperfine 'python tfp_sampler.py --samples 1000 --chains '"$CHAINS"' --runonce True --keep_all_samples False'
done
