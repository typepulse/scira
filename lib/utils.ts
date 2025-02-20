// /lib/utils.ts
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Globe, Book, YoutubeIcon, Pen, Mountain } from 'lucide-react';
import { Brain, Code, XLogo } from '@phosphor-icons/react';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type SearchGroupId = 'web' | 'academic' | 'analysis' | 'extreme';

export const searchGroups = [
  {
    id: 'web' as const,
    name: 'Web',
    description: 'Search across the entire internet',
    icon: Globe,
  },
  // {
  //   id: 'x' as const,
  //   name: 'X',
  //   description: 'Search X posts and content powered by Exa',
  //   icon: XLogo,
  // },
  {
    id: 'analysis' as const,
    name: 'Analysis',
    description: 'Code, stock and currency stuff',
    icon: Code,
  },
  {
    id: 'academic' as const,
    name: 'Academic',
    description: 'Search academic papers powered by Exa',
    icon: Book,
  },
  {
    id: 'extreme' as const,
    name: 'Extreme',
    description: 'Deep research with multiple sources and analysis',
    icon: Mountain,
  },
] as const;

export type SearchGroup = (typeof searchGroups)[number];
