// assets/matrix_audio.js
window.matrixAudio = {
    init: function() {
        this.bgm = new Howl({
            src: ['/assets/matrix_bgm.mp3'],
            loop: true,
            volume: 0.4,
            preload: true
        });
        this.glitch = new Howl({
            src: ['/assets/matrix_glitch.mp3'],
            volume: 0.6,
            preload: true
        });
    },
    startMatrixAudio: function() {
        if (this.bgm) this.bgm.play();
    },
    playGlitchEffect: function() {
        if (this.glitch) this.glitch.play();
    }
};

