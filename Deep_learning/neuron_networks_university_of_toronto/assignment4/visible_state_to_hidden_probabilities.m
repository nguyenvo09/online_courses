function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% rbm_w shape (no_hidden_units, num_visible_unit)
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
    [H, D] = size(rbm_w);
    [D, N] = size(visible_state);
    res = rand(H, N);
    r = rbm_w * visible_state;
    r = 1.0 ./ (1.0 + exp(-r));
    hidden_probability = r;
    %error('not yet implemented');
end
