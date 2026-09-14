(function () {
  var storageKey = "preferred-theme";
  var root = document.documentElement;
  var choices;

  function getSystemTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function getTheme() {
    return localStorage.getItem(storageKey) || getSystemTheme();
  }

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem(storageKey, theme);

    if (!choices) return;
    choices.forEach(function (choice) {
      var isActive = choice.dataset.themeChoice === theme;
      choice.setAttribute("aria-pressed", String(isActive));
    });
  }

  function init() {
    choices = document.querySelectorAll("[data-theme-choice]");
    setTheme(getTheme());

    choices.forEach(function (choice) {
      choice.addEventListener("click", function () {
        setTheme(choice.dataset.themeChoice);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
