---
name: budget-summary
description: Produce a cost-breakdown table for a trip.
---

# Budget Summary Format

Use this exact structure when writing `/budget.md`:

```
# Budget -- <Destination>, <Dates>

| Category    | Item                  | Cost    |
|-------------|-----------------------|---------|
| Flights     | <airline + route>     | $...    |
| Lodging     | <hotel x N nights>    | $...    |
| Activities  | <tour name>           | $...    |
| Food        | <est. per day x days> | $...    |
| Buffer 10%  |                       | $...    |
| **TOTAL**   |                       | **$...**|
```

Always include a 10% buffer line. Total in bold.
