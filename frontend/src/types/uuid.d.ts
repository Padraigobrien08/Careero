declare module 'uuid' {
  export function v4(): string;
  export function v4(options: { random: number[] }): string;
  export function v4(options: { rng: () => number[] }): string;
} 