# list all video files
directories, files, = glob_wildcards('/scratch/clemens10/playback/dat/{directory}/{file}.h264')

# remove hidden
directories = [directory for directory, file in zip(directories, files) if not file.startswith('.')]
files = [file for directory, file in zip(directories, files) if not file.startswith('.')]

# generate inputs and outputs from listing
mp4files = expand("/scratch/clemens10/playback/dat/{directory}/{file}.mp4", zip, directory=directories, file=files)
bgdfiles = expand("/scratch/clemens10/playback/dat/{directory}/{file}.png", zip, directory=directories, file=files)
trkfiles = expand("/scratch/clemens10/playback/dat/{directory}/{file}.h5", zip, directory=directories, file=files)
spdfiles = expand("/scratch/clemens10/playback/res/{directory}_spd.h5", zip, directory=directories, file=files)

rule all:
    input: mp4files, bgdfiles, trkfiles, spdfiles

rule containerize:
    input:  "/scratch/clemens10/playback/dat/{directory}/{videofile}.h264"
    output: "/scratch/clemens10/playback/dat/{directory}/{videofile}.mp4"
    log:    "/scratch/clemens10/playback/dat/{directory}/{videofile}_containerize.log"
    shell: "ffmpeg -i {input} -vcodec copy {output}"

rule estimate_background:
    input: "/scratch/clemens10/playback/dat/{directory}/{videofile}.mp4"
    output: "/scratch/clemens10/playback/dat/{directory}/{videofile}.png"
    log:    "/scratch/clemens10/playback/dat/{directory}/{videofile}_bg.log"
    params:
        format="png",
        nframes=1000
    shell: "python3 -m tracker.BackGround -n {params.nframes} -f {params.format} --savebin {input}"

rule track:
    input:
        video="/scratch/clemens10/playback/dat/{directory}/{videofile}.mp4",
        background="/scratch/clemens10/playback/dat/{directory}/{videofile}.tif"
    output: "/scratch/clemens10/playback/dat/{directory}/{videofile}.h5"
    log:    "/scratch/clemens10/playback/dat/{directory}/{videofile}_track.log"
    shell: "python  -m tracker.FlyPursuit {input.video} -t 0.25"

rule postprocess:
    input: "/scratch/clemens10/playback/dat/{directory}/{directory}.h5"
    output: "/scratch/clemens10/playback/res/{directory}_spd.h5"
    log:    "/scratch/clemens10/playback/dat/{directory}/{directory}_post.log"
    shell: "python3 postprocessing.py {input}"