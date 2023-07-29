module ESAAnalytics

# Import dependencies =========================================================

# Declare export ==============================================================

export shapleyvalueanalysis

"""
    shapleyvalueanalysis(benefits)

Calculate shapley values from benefits of all possible coalitions

`benefits` is a sequence of benefits from cooperating. The first element is the benefits when no one is in the coalition, and the last element is the benefits when all players coorperate. The output is a vector of contributions from respective participants.
"""
function shapleyvalueanalysis(benefits)
    nP = Int(log2(length(benefits)))
    map_idx_participation = [digits(Bool, idx, base=2, pad=nP) for idx ∈ 0:(length(benefits)-1)]
    converter = [2^i for i ∈ 0:(nP-1)]
    contribution = zeros(nP)
    for iP ∈ 1:nP, idx ∈ 1:2^nP
        if !map_idx_participation[idx][iP]
            nS = sum(map_idx_participation[idx])
            idx_with_iP = idx + converter[iP]
            contribution[iP] += factorial(nS)*factorial(nP - nS - 1)/factorial(nP) * (benefits[idx_with_iP] - benefits[idx])
        end
    end
    return contribution
end

end
