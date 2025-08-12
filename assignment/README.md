This is a trial project where in we are using gemma-2b-it to translate english text into spanish.
The main challenge here is deploying the binary onto a raspberry pi and running it onto that device.

We are using gemma.cpp library to access the model in C++. The code skeleton has been taken from the hello world example in gemma.cpp library.
Some changes have been made to the config and dr_wav.h library has been added to support transcribing audio files into text.
however the prompt for this was not working so the next steps would be troubleshooting this prompt and checking what can be done to make it work.
Another challenge in this task was making the cross compilation work witrh the cmake project.

This is a rough draft of PoC and the code needs to be modified for production readiness.

the following commands can be used to build the binary on any linux machine.
```sh
```

Make sure you delete the contents of the build directory before changing
configurations.

Then use `make` to build the project:

```sh
cd build
cmake -DBUILD_MODE=local ..
make hello_world
```

As with the top-level `gemma.cpp` project you can use the `make` commands `-j`
flag to use parallel threads for faster builds.

From inside the `gemma.cpp/examples/hello_world/build` directory, there should
be a `hello_world` executable. You can run it with the same 3 model arguments as
gemma.cpp specifying the tokenizer, compressed weights file, and model type, for
example:

```sh
./hello_world --tokenizer tokenizer.spm --compressed_weights 2b-it-sfp.sbs --model 2b-it
```


Next steps ::
- Troubleshoot transcribing and see how prompt enginerring can make transcribing happen
- Quantize the model to make the storage and memory consumption better
- restructure the code to include a wrapper over the gema model call so as to encapsulate it
