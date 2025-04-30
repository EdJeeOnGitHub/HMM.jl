using HMM
using Test

@testset "HMM.jl" begin
    # Test suite for SimpleHMM
    include("simple_hmm_tests.jl")

    # Test suite for MixtureHMM
    include("mixture_hmm_tests.jl")

    # Regression HMM tests
    include("regression_hmm_tests.jl")

    # Placeholder for future tests
    # include("utils_tests.jl")
end 