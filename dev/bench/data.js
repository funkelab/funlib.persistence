window.BENCHMARK_DATA = {
  "lastUpdate": 1795026420957,
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
      },
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
          "id": "86e78601aa5f83d773a6be183e55bf8eff97b005",
          "message": "Merge pull request #54 from funkelab/feat/csr-graph-database\n\nAdd a graph database in memory-mapped CSR arrays",
          "timestamp": "2026-11-18T18:23:46Z",
          "url": "https://github.com/funkelab/funlib.persistence/commit/86e78601aa5f83d773a6be183e55bf8eff97b005"
        },
        "date": 1795026333913,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import funlib.persistence (cold)",
            "value": 750.595,
            "range": "± 4.927",
            "unit": "ms",
            "extra": "interpreter startup 10.1 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 48.581,
            "unit": "s",
            "extra": "`uv run pytest` in the tests job"
          }
        ]
      },
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
          "id": "46255722eb353252c1f49ca78aeca6dfe29ec7d2",
          "message": "Merge pull request #52 from funkelab/fix/strict-metadata\n\nstrict_metadata=True raises when a metadata field is missing",
          "timestamp": "2026-11-18T18:25:23Z",
          "url": "https://github.com/funkelab/funlib.persistence/commit/46255722eb353252c1f49ca78aeca6dfe29ec7d2"
        },
        "date": 1795026420389,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import funlib.persistence (cold)",
            "value": 655.736,
            "range": "± 19.017",
            "unit": "ms",
            "extra": "interpreter startup 8.7 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 41.242,
            "unit": "s",
            "extra": "`uv run pytest` in the tests job"
          }
        ]
      }
    ]
  }
}