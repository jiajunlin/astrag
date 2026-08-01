<template>
  <section id="card">
    <h2>{{ title }}</h2>
  </section>
</template>

<script>
export default {
  props: { title: String },
  methods: {
    highlight() { this.active = true; }
  }
}
</script>

<style>
#card { border: 1px solid gray; }
</style>
