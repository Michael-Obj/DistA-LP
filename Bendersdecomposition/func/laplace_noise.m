% Function to generate Laplace noise
function noise = laplace_noise(scale, sz)
    uniform_random = rand(sz) - 0.5;
    noise = -scale * log(1 - 2 * abs(uniform_random));
end