import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from '@/components/ui/button';

describe('Button Component', () => {
  describe('Red Phase - Failing Tests', () => {
    it('should render with default props', () => {
      render(<Button>Click me</Button>);
      
      const button = screen.getByRole('button', { name: /click me/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveClass('bg-primary-600'); // Primary variant by default
      expect(button).toHaveClass('h-10'); // Medium size by default
    });

    it('should render different variants correctly', () => {
      const { rerender } = render(<Button variant="secondary">Secondary</Button>);
      let button = screen.getByRole('button');
      expect(button).toHaveClass('bg-secondary-100');

      rerender(<Button variant="outline">Outline</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveClass('border-secondary-300');

      rerender(<Button variant="ghost">Ghost</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveClass('text-secondary-700');

      rerender(<Button variant="destructive">Destructive</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveClass('bg-error-600');
    });

    it('should render different sizes correctly', () => {
      const { rerender } = render(<Button size="sm">Small</Button>);
      let button = screen.getByRole('button');
      expect(button).toHaveClass('h-8');

      rerender(<Button size="lg">Large</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveClass('h-12');

      rerender(<Button size="icon">Icon</Button>);
      button = screen.getByRole('button');
      expect(button).toHaveClass('h-10', 'w-10');
    });

    it('should handle disabled state', () => {
      render(<Button disabled>Disabled</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
      expect(button).toHaveClass('disabled:pointer-events-none', 'disabled:opacity-50');
    });

    it('should handle click events', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(<Button onClick={handleClick}>Click me</Button>);
      
      const button = screen.getByRole('button');
      await user.click(button);
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('should not trigger click when disabled', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(<Button disabled onClick={handleClick}>Disabled</Button>);
      
      const button = screen.getByRole('button');
      await user.click(button);
      
      expect(handleClick).not.toHaveBeenCalled();
    });

    it('should accept custom className', () => {
      render(<Button className="custom-class">Custom</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('custom-class');
      expect(button).toHaveClass('bg-primary-600'); // Should still have default classes
    });

    it('should forward ref correctly', () => {
      const ref = React.createRef<HTMLButtonElement>();
      render(<Button ref={ref}>Ref test</Button>);
      
      expect(ref.current).toBeInstanceOf(HTMLButtonElement);
      expect(ref.current).toHaveTextContent('Ref test');
    });

    it('should render as different HTML button types', () => {
      render(<Button type="submit">Submit</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('type', 'submit');
    });

    it('should have proper focus styles', () => {
      render(<Button>Focus test</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('focus-visible:outline-none');
      expect(button).toHaveClass('focus-visible:ring-2');
      expect(button).toHaveClass('focus-visible:ring-primary-500');
    });

    it('should have proper accessibility attributes', () => {
      render(
        <Button aria-label="Close dialog" aria-describedby="help-text">
          ×
        </Button>
      );
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'Close dialog');
      expect(button).toHaveAttribute('aria-describedby', 'help-text');
    });

    it('should support keyboard navigation', async () => {
      const handleClick = jest.fn();
      const user = userEvent.setup();
      
      render(<Button onClick={handleClick}>Keyboard test</Button>);
      
      const button = screen.getByRole('button');
      button.focus();
      expect(button).toHaveFocus();
      
      await user.keyboard('{Enter}');
      expect(handleClick).toHaveBeenCalledTimes(1);
      
      await user.keyboard(' ');
      expect(handleClick).toHaveBeenCalledTimes(2);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty children', () => {
      render(<Button />);
      
      const button = screen.getByRole('button');
      expect(button).toBeInTheDocument();
      expect(button).toHaveTextContent('');
    });

    it('should handle React node children', () => {
      render(
        <Button>
          <span data-testid="icon">🚀</span>
          <span>Launch</span>
        </Button>
      );
      
      const button = screen.getByRole('button');
      const icon = screen.getByTestId('icon');
      
      expect(button).toContainElement(icon);
      expect(button).toHaveTextContent('🚀Launch');
    });

    it('should maintain button behavior with custom props', () => {
      render(
        <Button
          variant="outline"
          size="lg"
          className="my-custom-class"
          data-testid="custom-button"
          aria-pressed="false"
        >
          Custom Button
        </Button>
      );
      
      const button = screen.getByTestId('custom-button');
      expect(button).toHaveClass('border-secondary-300'); // outline variant
      expect(button).toHaveClass('h-12'); // lg size
      expect(button).toHaveClass('my-custom-class'); // custom class
      expect(button).toHaveAttribute('aria-pressed', 'false');
    });
  });
});