% Reset environment to initial state and return initial observation
function masteragent = myReset(masteragent)
    masteragent.cuts_A = [];
    masteragent.cuts_b = [];
    masteragent.decision = [];
end