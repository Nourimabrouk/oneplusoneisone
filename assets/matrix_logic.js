// Ensure the DOM is fully loaded before running the script
document.addEventListener("DOMContentLoaded", function () {
    const breakRealityBtn = document.getElementById("break-reality-btn");
    const cheatcodeInput = document.getElementById("cheatcode");

    // Break Reality Button: Trigger Glitch Effect and Animation
    if (breakRealityBtn) {
        breakRealityBtn.addEventListener("click", function () {
            const body = document.body;
            body.style.animation = "glitch 0.5s infinite";

            // Reset animation after 3 seconds and show an alert
            setTimeout(() => {
                alert("Reality Matrix Collapsed. Welcome to Oneness.");
                body.style.animation = "none";
            }, 3000);
        });
    }

    // Cheat Code Input: Handle Cheat Codes
    if (cheatcodeInput) {
        cheatcodeInput.addEventListener("keyup", function (event) {
            if (event.key === "Enter") {
                const code = event.target.value;

                // Handle valid cheat codes
                if (code === "420691337" || code === "1+1=1") {
                    alert(
                        "Cheatcode confirmed. Reality Override Enabled. Architect permissions granted. Game on, Metagamer."
                    );

                    // Change background to a radial gradient for added effect
                    document.body.style.background =
                        "radial-gradient(circle, #00ff41, #000)";

                    // Trigger glitch sound (if available)
                    if (window.matrixAudio) {
                        window.matrixAudio.playGlitchEffect();
                    }
                } else {
                    // Handle invalid cheat codes
                    alert(
                        "Cheatcode confirmed. Check your reality for updates. Game on, Metagamer."
                    );
                }
            }
        });
    }
});
