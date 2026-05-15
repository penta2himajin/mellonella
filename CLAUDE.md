# mellonella — Claude Code session policy

## After creating a pull request

1. **Subscribe to the PR.** Immediately after `create_pull_request`
   returns, call `subscribe_pr_activity` with the new PR number. Webhook
   events (CI failures, review comments) will then wake the session so
   they can be triaged without prompting.

2. **Poll for completion.** Subscription only delivers *failure* events;
   it does **not** notify on green CI or on merge. When the goal is to
   know that a PR finished successfully, run a Monitor-backed poll loop
   that hits the GitHub API every 3 minutes (180 s) and exits when the
   PR's check runs all complete or the PR closes/merges. End the turn
   after starting the loop — the Monitor exit produces a notification.

3. **Unsubscribe only when told to.** If the user asks to stop watching
   a PR, call `unsubscribe_pr_activity` and stop pushing changes. Merge
   webhook events auto-unsubscribe.
