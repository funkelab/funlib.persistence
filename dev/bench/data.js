window.BENCHMARK_DATA = {
  "lastUpdate": 1794423822694,
  "repoUrl": "https://github.com/funkelab/funlib.persistence",
  "entries": {
    "funlib.persistence": [
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "12f49315db1345be43830f0cf0cfb8825efcd326",
          "message": "Merge pull request #48 from funkelab/docs-deploy-on-tags\n\nDeploy docs on version tags and by hand only",
          "timestamp": "2026-11-11T19:02:20Z",
          "url": "https://github.com/funkelab/funlib.persistence/commit/12f49315db1345be43830f0cf0cfb8825efcd326"
        },
        "date": 1794423822199,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import funlib.persistence (cold)",
            "value": 504.086,
            "range": "± 25.481",
            "unit": "ms",
            "extra": "interpreter startup 7.4 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 28.275,
            "unit": "s",
            "extra": "`uv run pytest` in the tests job"
          }
        ]
      }
    ]
  }
}