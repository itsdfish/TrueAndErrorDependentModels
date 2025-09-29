"""
    TEDM{T <: Real} <: AbstractTrueErrorModel{T}

A model object for a True and Error Model with depdence between choice sets in the same repetition. Two choice sets are presented twice during the same session, 
meaning 4 choices are made in total. Subscript r represents risky, subscript s represents safe, and subscripts 1 and 2
represent choice set. For example, `pᵣᵣ` represents the probability of truely prefering the risky option in both choice sets
and `ϵₛ₁` represents the error probability of choosing safe given a true preference for risky in first choice set. 

# Fields 

- `p::AbstractVector{T}`: a vector of true preference state probabilities with elements `p = [pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ]`, such that sum(p) = 1. 
- `ϵ::AbstractVector{T}`: a vector of error probabilities with elements `ϵ = [ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂]`.
- `p_rep::T``: the probability that the decision in the first choice set is repeated in the second choice set


# Constructors

TEDM(p, ϵ, p_rep)

TEDM(; p, ϵ, p_rep)

# Example 

```julia 
using TrueAndErrorDependentModels

dist = TEDM(; p = [0.60, .30, .05, .05], ϵ = fill(.10, 4), p_rep = 0)
data = rand(dist, 200)
logpdf(dist, data)
```

# References

Birnbaum, M. H., & Quispe-Torreblanca, E. G. (2018). TEMAP2. R: True and error model analysis program in R. Judgment and Decision Making, 13(5), 428-440.

Lee, M. D. (2018). Bayesian methods for analyzing true-and-error models. Judgment and Decision Making, 13(6), 622-635.
"""
struct TEDM{T <: Real} <: AbstractTrueErrorModel{T}
    p::AbstractVector{T}
    ϵ::AbstractVector{T}
    p_rep::T
end

function TEDM(p, ϵ, p_rep)
    _p, _ϵ = promote(p, ϵ)
    _, _p_rep = promote(p[1], p_rep)
    return TEDM(_p, _ϵ, _p_rep)
end

function TEDM(; p, ϵ, p_rep)
    return TEDM(p, ϵ, p_rep)
end

"""
    compute_probs(dist::TEDM{T})

Computes the joint probability for all 16 response categories
    
# Arguments

- `dist::TEDM{T}`: a distribution object for a True and Error Model for two choices sets, each containing
a risky option R and a safe option S.

# Output 

- `θ::Vector{T}`: vector of joint response probabilities with the following elements:

1.  RR,RR
2.  RR,RS
3.  RR,SR
4.  RR,SS
5.  RS,RR
6.  RS,RS
7.  RS,SR
8.  RS,SS
9.  SR,RR
10. SR,RS
11. SR,SR
12. SR,SS
13. SS,RR
14. SS,RS
15. SS,SR
16. SS,SS

where S corresponds to choosing the safe option, R corresponds to choosing the risky option, each pair (XX)
is the joint choice for choice sets 1 and two, respectively for a given replication. The first pair corresponds to 
the first replication, and the second pair corresponds to the second replication. For example, SR,RS indicates the selection 
of the safe option for choice set 1 followed by the risky option for choice set 2 during the first replication, and the 
reversal of choices for the second replication. 
"""
function compute_probs(dist::TEDM{T}) where {T}
    (; p, ϵ, p_rep) = dist
    pᵣᵣ, pᵣₛ, pₛᵣ, pₛₛ = p
    ϵₛ₁, ϵₛ₂, ϵᵣ₁, ϵᵣ₂ = ϵ

    θ = zeros(T, 16)

    # RR,RR
    θ[1] =
        pᵣᵣ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - ϵₛ₁) *
        (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) +
        pᵣₛ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - ϵₛ₁) *
        (p_rep + (1 - p_rep) * ϵᵣ₂) +
        pₛᵣ * ϵᵣ₁ * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * ϵᵣ₁ *
        (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) +
        pₛₛ * ϵᵣ₁ * (p_rep + (1 - p_rep) * ϵᵣ₂) * ϵᵣ₁ * (p_rep + (1 - p_rep) * ϵᵣ₂)
    # RR,RS
    θ[2] =
        pᵣᵣ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - p_rep) * (1 - ϵₛ₁) *
        ϵₛ₂ +
        pᵣₛ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - p_rep) * (1 - ϵₛ₁) *
        (1 - ϵᵣ₂) +
        pₛᵣ * ϵᵣ₁ * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - p_rep) * ϵᵣ₁ * ϵₛ₂ +
        pₛₛ * ϵᵣ₁ * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - p_rep) * ϵᵣ₁ * (1 - ϵᵣ₂)
    # RR,SR
    θ[3] =
        pᵣᵣ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - p_rep) * ϵₛ₁ *
        (1 - ϵₛ₂) +
        pᵣₛ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - p_rep) * ϵₛ₁ * ϵᵣ₂ +
        pₛᵣ * ϵᵣ₁ * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - p_rep) * (1 - ϵᵣ₁) *
        (1 - ϵₛ₂) +
        pₛₛ * ϵᵣ₁ * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - p_rep) * (1 - ϵᵣ₁) * ϵᵣ₂
    # RR,SS
    θ[4] =
        pᵣᵣ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * ϵₛ₁ *
        (p_rep + (1 - p_rep) * ϵₛ₂) +
        pᵣₛ * (1 - ϵₛ₁) * (p_rep + (1 - p_rep) * ϵᵣ₂) * ϵₛ₁ *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) +
        pₛᵣ * ϵᵣ₁ * (p_rep + (1 - p_rep) * (1 - ϵₛ₂)) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * ϵₛ₂) +
        pₛₛ * ϵᵣ₁ * (p_rep + (1 - p_rep) * ϵᵣ₂) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂))
    # RS,RR
    θ[5] = θ[2]
    # RS,RS
    θ[6] =
        pᵣᵣ * (1 - ϵₛ₁) * (1 - p_rep) * ϵₛ₂ * (1 - p_rep) * (1 - ϵₛ₁) * ϵₛ₂ +
        pᵣₛ * (1 - ϵₛ₁) * (1 - p_rep) * (1 - ϵᵣ₂) * (1 - p_rep) * (1 - ϵₛ₁) * (1 - ϵᵣ₂) +
        pₛᵣ * ϵᵣ₁ * (1 - p_rep) * ϵₛ₂ * (1 - p_rep) * ϵᵣ₁ * ϵₛ₂ +
        pₛₛ * ϵᵣ₁ * (1 - p_rep) * (1 - ϵᵣ₂) * (1 - p_rep) * ϵᵣ₁ * (1 - ϵᵣ₂)
    # RS,SR
    θ[7] =
        pᵣᵣ * (1 - ϵₛ₁) * (1 - p_rep) * ϵₛ₂ * (1 - p_rep) * ϵₛ₁ * (1 - ϵₛ₂) +
        pᵣₛ * (1 - ϵₛ₁) * (1 - p_rep) * (1 - ϵᵣ₂) * (1 - p_rep) * ϵₛ₁ * ϵᵣ₂ +
        pₛᵣ * ϵᵣ₁ * (1 - p_rep) * ϵₛ₂ * (1 - p_rep) * (1 - ϵᵣ₁) * (1 - ϵₛ₂) +
        pₛₛ * ϵᵣ₁ * (1 - p_rep) * (1 - ϵᵣ₂) * (1 - p_rep) * (1 - ϵᵣ₁) * ϵᵣ₂
    # RS,SS
    θ[8] =
        pᵣᵣ * (1 - ϵₛ₁) * (1 - p_rep) * ϵₛ₂ * ϵₛ₁ * (p_rep + (1 - p_rep) * ϵₛ₂) +
        pᵣₛ * (1 - ϵₛ₁) * (1 - p_rep) * (1 - ϵᵣ₂) * ϵₛ₁ *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) +
        pₛᵣ * ϵᵣ₁ * (1 - p_rep) * ϵₛ₂ * (1 - ϵᵣ₁) * (p_rep + (1 - p_rep) * ϵₛ₂) +
        pₛₛ * ϵᵣ₁ * (1 - p_rep) * (1 - ϵᵣ₂) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂))
    # SR,RR
    θ[9] = θ[3]
    # SR,RS
    θ[10] = θ[7]
    # SR,SR
    θ[11] =
        pᵣᵣ * ϵₛ₁ * (1 - p_rep) * (1 - ϵₛ₂) * ϵₛ₁ * (1 - p_rep) * (1 - ϵₛ₂) +
        pᵣₛ * ϵₛ₁ * (1 - p_rep) * ϵᵣ₂ * ϵₛ₁ * (1 - p_rep) * ϵᵣ₂ +
        pₛᵣ * (1 - ϵᵣ₁) * (1 - p_rep) * (1 - ϵₛ₂) * (1 - ϵᵣ₁) * (1 - p_rep) * (1 - ϵₛ₂) +
        pₛₛ * (1 - ϵᵣ₁) * (1 - p_rep) * ϵᵣ₂ * (1 - ϵᵣ₁) * (1 - p_rep) * ϵᵣ₂
    # SR,SS
    θ[12] =
        pᵣᵣ * ϵₛ₁ * (1 - p_rep) * (1 - ϵₛ₂) * ϵₛ₁ * (p_rep + (1 - p_rep) * ϵₛ₂) +
        pᵣₛ * ϵₛ₁ * (1 - p_rep) * ϵᵣ₂ * ϵₛ₁ * (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) +
        pₛᵣ * (1 - ϵᵣ₁) * (1 - p_rep) * (1 - ϵₛ₂) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * ϵₛ₂) +
        pₛₛ * (1 - ϵᵣ₁) * (1 - p_rep) * ϵᵣ₂ * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂))
    # SS,RR
    θ[13] = θ[4]
    # SS,RS
    θ[14] = θ[8]
    # SS,SR
    θ[15] = θ[12]
    # SS,SS
    θ[16] =
        pᵣᵣ * ϵₛ₁ * (p_rep + (1 - p_rep) * ϵₛ₂) * ϵₛ₁ * (p_rep + (1 - p_rep) * ϵₛ₂) +
        pᵣₛ * ϵₛ₁ * (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) * ϵₛ₁ *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) +
        pₛᵣ * (1 - ϵᵣ₁) * (p_rep + (1 - p_rep) * ϵₛ₂) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * ϵₛ₂) +
        pₛₛ * (1 - ϵᵣ₁) * (p_rep + (1 - p_rep) * (1 - ϵᵣ₂)) * (1 - ϵᵣ₁) *
        (p_rep + (1 - p_rep) * (1 - ϵᵣ₂))
    return θ
end
