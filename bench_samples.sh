set +e

for SAMPLES in 100 1000 10000
do
	hyperfine 'python jax_sampler.py --samples '"$SAMPLES"' --chains 1000'
        hyperfine 'python tf_sampler.py --samples '"$SAMPLES"' --chains 1000 --runonce True'
        hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000 --runonce True'
        hyperfine 'python tf_sampler.py --samples '"$SAMPLES"' --chains 1000 --runonce True --keep_all_samples False'
        hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000 --runonce True --keep_all_samples False'
done
