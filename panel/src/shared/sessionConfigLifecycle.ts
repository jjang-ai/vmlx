/** A sleeping local engine still owns its socket and launch-time configuration. */
export function hasLiveLocalSession(session: { status: string; type?: string }): boolean {
  return session.type !== 'remote' && ['running', 'loading', 'standby'].includes(session.status);
}

/** Keep launch-time values effective until an actual restart; save future values separately. */
export function planSessionConfigSave(
  session: { status: string; type?: string },
  effective: Record<string, unknown>,
  desired: Record<string, unknown>,
  restartKeys: ReadonlySet<string>,
) {
  const changedKeys = [...restartKeys].filter(key =>
    JSON.stringify(effective[key]) !== JSON.stringify(desired[key]));
  const restartRequired = hasLiveLocalSession(session) && changedKeys.length > 0;
  const config = { ...desired };
  if (restartRequired) {
    for (const key of restartKeys) {
      if (Object.prototype.hasOwnProperty.call(effective, key)) config[key] = effective[key];
      else delete config[key];
    }
  }
  return { config, pendingConfig: restartRequired ? { ...desired } : null, restartRequired, changedKeys };
}
