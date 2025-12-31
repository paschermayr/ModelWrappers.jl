############################################################################################
# io.jl - Utility functions for package development with environment management
############################################################################################

using Pkg

############################################################################################
# Package Paths Configuration
############################################################################################

const PACKAGES_BASE_DIR = "D:\\Patrick\\OneDrive\\1_Life\\1_Git\\Code\\Julia\\packages"

const PACKAGE_PATHS = Dict(
    "ModelWrappers" => joinpath(PACKAGES_BASE_DIR, "ModelWrappers.jl"),
    "BaytesCore" => joinpath(PACKAGES_BASE_DIR, "BaytesCore.jl"),
)

############################################################################################
# Environment Information
############################################################################################

"""
    show_environment_info()

Display detailed information about the current environment and loaded packages.
"""
function show_environment_info()
    println("\n" * "="^80)
    println("Environment Information")
    println("="^80)
    
    # Active project
    active_proj = Base.active_project()
    if active_proj !== nothing
        println("\n📁 Active Project:")
        println("   $(active_proj)")
    else
        println("\n📁 Active Project: None (using global environment)")
    end
    
    # Environment directory
    println("\n📂 Environment Directory:")
    println("   $(dirname(active_proj))")
    
    # Check if Project.toml exists
    if isfile(active_proj)
        println("\n✓ Project.toml exists")
        
        # Read and show dependencies
        project = Pkg.TOML.parsefile(active_proj)
        if haskey(project, "deps")
            println("\n📦 Dependencies in Project.toml:")
            for (name, uuid) in project["deps"]
                println("   • $name")
            end
        end
    else
        println("\n⚠ Project.toml does not exist yet")
        println("   Use `] add PackageName` to create it")
    end
    
    # Show where packages actually load from
    println("\n🔍 Package Load Locations:")
    for (pkg_name, _) in PACKAGE_PATHS
        try
            mod = Base.require(Main, Symbol(pkg_name))
            path = pathof(mod)
            
            if occursin(PACKAGES_BASE_DIR, path)
                println("   ✓ $pkg_name: [LOCAL DEV]")
            else
                println("   ⚠ $pkg_name: [REGISTRY]")
            end
            println("     → $path")
        catch
            println("   • $pkg_name: [Not loaded yet]")
        end
    end
    
    println("\n" * "="^80 * "\n")
end

############################################################################################
# Core Functions - Switch between local and registry versions
############################################################################################

"""
    switch_to_local(package_name::String)

Switch a package to use the local development version IN THE CURRENT ENVIRONMENT.
"""
function switch_to_local(package_name::String)
    if !haskey(PACKAGE_PATHS, package_name)
        @warn "Package '$package_name' not found in PACKAGE_PATHS."
        return false
    end
    
    local_path = PACKAGE_PATHS[package_name]
    
    if !isdir(local_path)
        @error "Local path does not exist: $local_path"
        return false
    end
    
    println("Switching '$package_name' to local development version...")
    println("  Environment: $(Base.active_project())")
    println("  Local Path: $local_path")
    
    try
        Pkg.develop(path=local_path)
        println("✓ Successfully switched '$package_name' to local dev mode")
        println("⚠ IMPORTANT: Restart Julia to reload the package!")
        return true
    catch e
        @error "Failed to switch '$package_name' to local dev" exception=e
        return false
    end
end

"""
    switch_to_registry(package_name::String)

Switch a package back to the registry version IN THE CURRENT ENVIRONMENT.
"""
function switch_to_registry(package_name::String)
    println("Switching '$package_name' to registry version...")
    println("  Environment: $(Base.active_project())")
    
    try
        Pkg.free(package_name)
        Pkg.add(package_name)
        println("✓ Successfully switched '$package_name' to registry version")
        println("⚠ IMPORTANT: Restart Julia to reload the package!")
        return true
    catch e
        @error "Failed to switch '$package_name' to registry" exception=e
        return false
    end
end

############################################################################################
# Batch Operations
############################################################################################

"""
    use_local_dev()

Switch all configured packages to local development versions.
"""
function use_local_dev()
    println("\n" * "="^80)
    println("Switching ALL packages to local development versions")
    println("Environment: $(Base.active_project())")
    println("="^80 * "\n")
    
    for (pkg_name, _) in PACKAGE_PATHS
        switch_to_local(pkg_name)
        println()
    end
    
    println("="^80)
    println("✓ All packages switched to local dev mode!")
    println("⚠ IMPORTANT: You must restart Julia for changes to take effect!")
    println("="^80)
end

"""
    use_registry_versions()

Switch all configured packages back to registry versions.
"""
function use_registry_versions()
    println("\n" * "="^80)
    println("Switching ALL packages to registry versions")
    println("Environment: $(Base.active_project())")
    println("="^80 * "\n")
    
    for (pkg_name, _) in PACKAGE_PATHS
        switch_to_registry(pkg_name)
        println()
    end
    
    println("="^80)
    println("✓ All packages switched to registry mode!")
    println("⚠ IMPORTANT: You must restart Julia for changes to take effect!")
    println("="^80)
end

############################################################################################
# Status Functions
############################################################################################

"""
    show_package_status(package_name::String)

Display the current status of a package in the current environment.
"""
function show_package_status(package_name::String)
    try
        pkg_info = Pkg.dependencies()
        
        pkg_uuid = nothing
        for (uuid, info) in pkg_info
            if info.name == package_name
                pkg_uuid = uuid
                break
            end
        end
        
        if pkg_uuid === nothing
            println("  $package_name: NOT INSTALLED in this environment")
            return
        end
        
        info = pkg_info[pkg_uuid]
        
        print("  $package_name: ")
        
        if info.is_tracking_path
            println("[DEV] ✓")
            println("    → $(info.source)")
        elseif info.is_tracking_repo
            println("[GIT]")
            println("    → $(info.source)")
        else
            println("[REGISTRY] v$(info.version)")
        end
        
    catch e
        println("  $package_name: Error - $e")
    end
end

"""
    list_dev_packages()

List all packages in development mode in the current environment.
"""
function list_dev_packages()
    println("\n" * "="^80)
    println("Packages in Development Mode")
    println("Environment: $(Base.active_project())")
    println("="^80)
    
    pkg_info = Pkg.dependencies()
    dev_packages = []
    
    for (uuid, info) in pkg_info
        if info.is_tracking_path
            push!(dev_packages, (info.name, info.source))
        end
    end
    
    if isempty(dev_packages)
        println("  No packages in dev mode")
    else
        for (name, path) in sort(dev_packages, by=x->x[1])
            println("  ✓ $name")
            println("    → $path")
        end
    end
    
    println("="^80 * "\n")
end

"""
    verify_local_usage()

Verify that local versions are actually being used.
"""
function verify_local_usage()
    println("\n" * "="^80)
    println("Verifying Package Sources")
    println("="^80 * "\n")
    
    for (pkg_name, expected_path) in PACKAGE_PATHS
        try
            mod = Base.require(Main, Symbol(pkg_name))
            actual_path = pathof(mod)
            
            if occursin(PACKAGES_BASE_DIR, actual_path)
                println("✓ $pkg_name: Using LOCAL version")
                println("  → $actual_path")
            else
                println("⚠ $pkg_name: Using REGISTRY version")
                println("  → $actual_path")
                println("  Expected: $expected_path")
            end
            println()
        catch e
            println("✗ $pkg_name: Not loaded - $e")
            println()
        end
    end
    
    println("="^80 * "\n")
end


# Check specific package
function check_package_version(pkg_name::String)
    deps = Pkg.dependencies()
    for (uuid, info) in deps
        if info.name == pkg_name
            if info.is_tracking_path
                println("✓ $pkg_name: [LOCAL DEV]")
                println("  Path: $(info.source)")
            elseif info.is_tracking_repo
                println("✓ $pkg_name: [GIT REPO]")
                println("  Repo: $(info.source)")
            else
                println("✓ $pkg_name: [REGISTRY]")
                println("  Version: v$(info.version)")
            end
            return
        end
    end
    println("✗ $pkg_name: NOT INSTALLED")
end

############################################################################################
# Helper Functions
############################################################################################

"""
    show_configured_packages()

Display all configured package paths.
"""
function show_configured_packages()
    println("\n" * "="^80)
    println("Configured Package Paths")
    println("="^80)
    
    for (name, path) in sort(collect(PACKAGE_PATHS), by=x->x[1])
        exists = isdir(path) ? "✓" : "✗"
        println("  $exists $name")
        println("    → $path")
    end
    
    println("="^80 * "\n")
end

"""
    quick_setup()

Quick setup for development environment.
"""
function quick_setup()
    println("\n🚀 Quick Development Setup\n")
    show_environment_info()
    use_local_dev()
    println("\n⚠ Please restart Julia and run include(\"main.jl\") again")
end




# Export functions
export switch_to_local, switch_to_registry, check_package_version
export use_local_dev, use_registry_versions
export show_package_status, list_dev_packages
export show_environment_info, verify_local_usage
export show_configured_packages, quick_setup