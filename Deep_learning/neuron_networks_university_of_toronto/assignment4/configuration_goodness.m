function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    [H, D] = size(rbm_w);
    [D, N] = size(visible_state);
    [H, N] = size(hidden_state);
    r = hidden_state'*rbm_w; %shape (N,D)
    r = r' .* visible_state; %shape (D,N)
    r = sum(r, 1); %shape (1, N);
    G = 1.0 / N * sum(r);
    %error('not yet implemented');
end
