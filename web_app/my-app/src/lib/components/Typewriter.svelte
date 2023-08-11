
<script>
  export let text;
  export let fontSize;
  export let typingSpeed = 100;

  let displayedText = '';

  let charIndex = 0;
  let cursorVisible = true;
  function startTyping() {
    if (charIndex < text.length) {
      displayedText += text[charIndex];
      charIndex++;

      if (cursorVisible) {
        cursorVisible = false;
      } else {
        cursorVisible = true;
      }

      setTimeout(startTyping, typingSpeed);
    }
  }

  startTyping();
</script>

<div class="typewriter-container">
  <span
          class="typewriter-text neon-gradient"
          style="font-size: {fontSize};"
  >
    {displayedText}
  </span>
  <span class="cursor" class:blinking={cursorVisible}></span>
</div>

<style>
  .typewriter-container {
    position: relative;
  }

  .typewriter-text {
    display: inline-block;
  }

  .cursor {
    display: inline-block;
    width: 0.15em;
    height: 1.1em;
    background-color: #ffffff;
    opacity: 0.8;
    margin-left: 0.1em;
  }

  .blinking {
    animation: blink-animation 0.7s step-end infinite;
  }
  .neon-gradient {
    background: linear-gradient(45deg, #006dff, #ff7300);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  @keyframes blink-animation {
    0%, 100% {
      opacity: 0.8;
    }
    50% {
      opacity: 0;
    }
  }
</style>
