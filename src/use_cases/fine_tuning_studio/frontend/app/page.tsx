'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { CircularProgress, Box } from '@mui/material';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to dashboard on home page load
    router.push('/dashboard');
  }, [router]);

  return (
    <Box
      display='flex'
      justifyContent='center'
      alignItems='center'
      minHeight='100vh'
    >
      <CircularProgress />
    </Box>
  );
}