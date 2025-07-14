# lorenz

Back in 2011, I wanted to generate an animation of the Lorenz attractor, a famous
mathematical construct that demonstrates long term chaotic behavior.  I documented
it in [this (short) blog post](https://brainwagon.org/blog/my-own-animation-of-the-lorenz-attractor/)
along with a link to a youtube video.

But it worked in a pretty naive and stupid way.  First of all, it was just a command line 
program that generated images that I would later assemble using `ffmpeg` or some other program.
This means it was pretty tedious to experiment with.

It also works in a pretty silly way.  It uses Runge Kutte integration to march particles around
by very slow steps.  At each steps, it adds 1 to a temporary buffer at points surrounding the target
point with a gaussian distribution.  This was done because I was frankly too lazy to do the actual
math to compute the integrals over a pixel neighborhood.  Never got around to doing that, but perhaps
in a future version.

Lately I've been uploading some of this older code, and in trying to understand how far AI coding
assistants have progressed, converted it to different languages.  I thought that the 
[taichi extensions to the Python language](https://www.taichi-lang.org/) would be appropriate,
since it can compile code to execute onto a GPU if you have one, but can also write highly
efficient SIMD kernels for more conventional processors.  I handed my code to the copilot AI in 
github, and it gave it a try.

It took several tries, with me pushing the AI in certain directions.  It made a few mistakes about
trying to call taichi functions from Python code, and other miscellaneous problems.  And in the end, 
it still is using numpy to handle the normalization of colors and the like.  For reasons which 
I don't fully understand (I am not that experienced with taichi or with GPU programming in general, 
which is one of the reasons that I am embarking on this experiment).  In CPU mode, I can get 
60fps, but probably more like 40fps on the NVIDIA gpu I have (an RTX 4060) which should be a lot 
faster.  I might tinker with this some more to try to understand what's going on.

## Update

I fixed numerous issues, and made lots of improvements.  Now I get 60fps in either CPU or FPU mode on 
my desktop, whereas my wimpy laptop gets more like 40fps in CPU mode and even less in GPU mode (does it
even have a GPU?)   It can also run in full screen mode now.
