<!-- +layout.svelte -->

<main>
    <nav class="navbar">
        <div class="navbar-left">
            <a href="/">
                <img src="favicon.png" alt="Home" class="home-button" />
            </a>
        </div>
        <div class="navbar-right">
            <a href="/classical">Classical</a>
            <a href="/quantum">Quantum</a>
        </div>
    </nav>

    <!-- Content of your pages will go here -->
    <slot></slot>
</main>

<style>
    /* Dark-themed styling for the navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #333;
        padding: 1rem;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
    }

    .navbar-left {
        display: flex;
        align-items: center;
    }

    .home-button {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }

    .navbar-right {
        display: flex;
    }

    .navbar a {
        color: #fff;
        text-decoration: none;
        margin: 0 30px 0 0;
        font-weight: bold;
        border: 2px solid transparent;
        padding: 5px 10px;
        transition: border-color 0.3s;
    }

    .navbar a:hover {
        border-color: #00ff00; /* Change to the desired color on hover */
    }
</style>
