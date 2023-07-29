module ESAAnalytics

# Import dependencies =========================================================
import MultivariateStats
import UMAP

# Declare export ==============================================================
export project_pca, project_umap
export shapleyvalueanalysis

"""
    project_pca(data::Matrix; dims=2)

Project `data`, a matrix of (features x observation), into a lower dimension space using the PCA method

# Keyword Arguments
- `dims` : number of dimensions of the projections
"""
function project_pca(data::Matrix; dims=2)
    model = MultivariateStats.fit(MultivariateStats.PCA, data; maxoutdim=dims)
    explainedvar = round(model.tprinvar / model.tvar, digits=4)
    @info "Explained variance $(explainedvar)"
    projecteddata = MultivariateStats.predict(model, data)
    return projecteddata
end

"""
    project_umap(data::Matrix; dims=2, kwargs...)

Project `data`, a matrix of (features x observation), into a lower dimension space using the UMAP method

# Keyword Arguments
- `dims` : number of dimensions of the projections
- `n_neighbors::Int=15` : This controls how many neighbors around each point are considered to be part of its local neighborhood. Larger values will result in embeddings that capture more global structure, while smaller values will preserve more local structures.
- `min_dist::Float=0.1` : This controls the minimum spacing of points in the embedding. Larger values will cause points to be more evenly distributed, while smaller values will preserve more local structure.

See also : [`UMAP.UMAP_`](@ref)
"""
function project_umap(data::Matrix; dims=2, kwargs...)
    model = UMAP.UMAP_(data, dims; kwargs...)
    projdata = model.embedding
    return projdata
end

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
