-- A solution contains projects, and defines the available configurations
--
solution "texture-clean"

	-- global config
	language "C++"
	configurations { "debug" }
	location "build"
	includedirs { "../include", "/usr/local/include", "/usr/include" }
	libdirs { "/usr/lib", "/usr/local/lib" }
	flags { "Symbols", "FatalWarnings", "ExtraWarnings" }

	-- Stack traces
	linkoptions { "-rdynamic" }

	-- Eigen
	includedirs { "/usr/include/eigen3" }

	-- debug: make config=debug
	configuration { "debug" }
		kind "ConsoleApp"
		defines { "DEBUG" }
		buildoptions { "-g3", "-O0" }
		targetdir "build/debug"

	project "densecrf_memory_test"
		files {
            "memory_test.cpp",
			"../include/*.h",
            "../src/densecrf.cpp",
            "../src/labelcompatibility.cpp",
            "../src/pairwise.cpp",
            "../src/permutohedral.cpp",
            "../src/unary.cpp",
            "../src/util.cpp",
            "../src/densecrf_map.cpp"
		}
