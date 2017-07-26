function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    
    %For question 9 only 
    visible_data = sample_bernoulli(visible_data);
    
    [H, D] = size(rbm_w);
    [D,N] = size(visible_data);
    
    vi2hid_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    %we'll sample a binary state for the hidden units conditional on the data
    binary_hid1 = sample_bernoulli(vi2hid_prob);
    hid2vi_prob = hidden_state_to_visible_probabilities(rbm_w, binary_hid1);
    %we need this part 
    % we'll sample a binary state for the visible units conditional on that binary hidden state 
    %(this is sometimes called the "reconstruction" for the visible units);
    binary_vis1_reconstruction = sample_bernoulli(hid2vi_prob);
    binary_hid2 = visible_state_to_hidden_probabilities(rbm_w, binary_vis1_reconstruction);
    
    %we'll sample a binary state for the hidden units conditional on 
    % that binary visible "reconstruction" state
    %binary_hid1_after_sample_from_reconstruction = sample_bernoulli(binary_hid2);
    %error('not yet implemented');
    
    %positive
    r1 = configuration_goodness_gradient(visible_data, binary_hid1);
    %negative q7
    %r2 = configuration_goodness_gradient(binary_vis1_reconstruction, binary_hid1_after_sample_from_reconstruction);
    r2 = configuration_goodness_gradient(binary_vis1_reconstruction, binary_hid2);

    
    ret = r1 - r2;
end
