using ModelWrappers
using Documenter

DocMeta.setdocmeta!(ModelWrappers, :DocTestSetup, :(using ModelWrappers); recursive=true)

makedocs(;
    modules=[ModelWrappers],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/ModelWrappers.jl/blob/{commit}{path}#{line}",
    sitename="ModelWrappers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/ModelWrappers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/ModelWrappers.jl",
    devbranch="main",
)
