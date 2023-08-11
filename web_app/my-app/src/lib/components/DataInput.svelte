
<script>
    export let title;
    export let description;
    export let inputType;
    export let options = [];
</script>

<div class="data-input">
    <h2>{title}</h2>
    <p>{description}</p>

    {#if inputType === 'number'}
        <input type="number" />
    {:else if inputType === 'dropdown'}
        <select>
            {#each options as option}
                <option value={option}>{option}</option>
            {/each}
        </select>
    {/if}
</div>

<style>
    .data-input {
        background-color: #333;
        color: #fff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    input,
    select {
        background-color: #444;
        color: #fff;
        padding: 0.5rem;
        border: none;
        border-radius: 4px;
        width: 100%;
    }
</style>
